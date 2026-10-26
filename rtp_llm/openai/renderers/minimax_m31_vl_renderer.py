"""MiniMax-M3.1 vision-language chat input renderer."""

import copy
import json

from jinja2 import pass_eval_context

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import ThinkingMode
from rtp_llm.config.response_format import normalize_think_tag
from rtp_llm.openai.api_datatype import ContentPartTypeEnum, MMPreprocessConfigPart
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import BasicRenderer
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs
from rtp_llm.openai.renderers.llava_renderer import get_preprocess_config
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
    ReasoningToolStreamStatus,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.minimax_m31_detector import (
    MiniMaxM31Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import ReasoningParser
from rtp_llm.utils.base_model_datatypes import MMUrlType


class MiniMaxM31Renderer(BasicRenderer, ReasoningToolBaseRenderer):
    def __init__(
        self,
        tokenizer,
        renderer_params,
        generate_env_config,
        render_config=None,
        ckpt_path=None,
        misc_config=None,
        vit_config=None,
    ):
        # The official M3.1 template always leaves the answer in adaptive mode.
        generate_env_config = copy.copy(generate_env_config)
        if (
            generate_env_config.think_mode == "disabled"
            and not generate_env_config.think_mode_explicit
        ):
            generate_env_config.think_mode = "adaptive"
        if normalize_think_tag(generate_env_config.think_start_tag) == "<think>\n":
            generate_env_config.think_start_tag = "<mm:think>"
        if normalize_think_tag(generate_env_config.think_end_tag) == "</think>\n\n":
            generate_env_config.think_end_tag = "</mm:think>"
        super().__init__(
            tokenizer,
            renderer_params,
            generate_env_config,
            render_config,
            ckpt_path,
            misc_config,
            vit_config,
        )

    def resolve_thinking_mode(self, request):
        # The released template defaults to adaptive, while explicit request
        # modes follow main's request contract.
        return request.resolve_thinking_mode(self.default_thinking_mode)

    def should_process_think(self, request):
        # ReasoningToolBaseRenderer already separates the MiniMax think tags.
        return False

    def _create_detector(self, request):
        return MiniMaxM31Detector() if request.tools else None

    def _create_reasoning_parser(self, request):
        mode = self.resolve_thinking_mode(request)
        if mode == ThinkingMode.DISABLED:
            return None
        parser = ReasoningParser(
            "minimax_m31", force_reasoning=mode == ThinkingMode.ENABLED
        )
        parser.detector.think_start_token = self.think_start_tag
        parser.detector.think_end_token = self.think_end_tag
        return parser

    def _compile_jinja_template(self, chat_template):
        template = super()._compile_jinja_template(chat_template)

        @pass_eval_context
        def tojson(_context, value, **kwargs):
            kwargs.setdefault("sort_keys", False)
            kwargs.setdefault("ensure_ascii", False)
            return json.dumps(value, **kwargs)

        template.environment.filters["tojson"] = tojson
        return template.environment.from_string(chat_template)

    async def _flush_buffer(
        self, buffer_list, stop_words_str, is_streaming, think_status_list
    ):
        pending_reasoning = []
        for status in buffer_list:
            reasoning, normal = "", ""
            if isinstance(status, ReasoningToolStreamStatus):
                if status.reasoning_parser is not None:
                    reasoning, normal = (
                        status.reasoning_parser.detector.drain_pending_text()
                    )
                if isinstance(status.detector, MiniMaxM31Detector):
                    status.delta_output_string += (
                        status.detector.drain_pending_normal_text()
                    )
                status.delta_output_string += normal
            pending_reasoning.append(reasoning)
        response = await super()._flush_buffer(
            buffer_list, stop_words_str, is_streaming, think_status_list
        )
        for choice, reasoning in zip(response.choices, pending_reasoning):
            if reasoning:
                choice.delta.reasoning_content = (
                    choice.delta.reasoning_content or ""
                ) + reasoning
        return response

    def _render_request(self, request, request_dict, urls=(), types=(), configs=()):
        for message in request_dict["messages"]:
            for call in message.get("tool_calls") or ():
                function = call.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        function["arguments"] = json.loads(arguments or "{}")
                    except json.JSONDecodeError:
                        raise FtRuntimeException(
                            ExceptionType.INVALID_PARAMS,
                            "MiniMax-M3.1 historical tool arguments must be valid JSON",
                        )
        render_args = {
            "messages": request_dict["messages"],
            "json": json,
            "add_generation_prompt": self.add_generation_prompt,
        }
        render_args.update(self.special_tokens_map)
        if request_dict.get("functions"):
            render_args["functions"] = request_dict["functions"]
        if request_dict.get("tools"):
            render_args["tools"] = request_dict["tools"]
        if request.chat_template_kwargs is not None:
            render_args.update(request.chat_template_kwargs)
        if request.extra_configs is not None and isinstance(
            request.extra_configs.chat_template_kwargs, dict
        ):
            render_args.update(request.extra_configs.chat_template_kwargs)
        prompt = self._get_template(request).render(**render_args)
        if render_args["add_generation_prompt"]:
            thinking_mode = self.resolve_thinking_mode(request)
            if thinking_mode == ThinkingMode.DISABLED:
                prompt += "</mm:think>"
            elif thinking_mode == ThinkingMode.ENABLED:
                prompt += "<mm:think>"
        return RenderedInputs(
            input_ids=self.tokenizer.encode(prompt),
            rendered_prompt=prompt,
            input_urls=list(urls),
            input_urls_type=list(types),
            preprocess_configs=list(configs),
        )

    def render_chat(self, request):
        request_dict = json.loads(request.model_dump_json(exclude_none=True))
        return self._render_request(request, request_dict)


class MiniMaxM31VLRenderer(MiniMaxM31Renderer):
    def render_chat(self, request):
        request_dict = json.loads(request.model_dump_json(exclude_none=True))
        urls, types, preprocess_configs = [], [], []
        for message, rendered_message in zip(
            request.messages, request_dict["messages"]
        ):
            if not isinstance(message.content, list):
                continue
            content = []
            for part in message.content:
                if part.type == ContentPartTypeEnum.text:
                    content.append({"type": "text", "text": part.text})
                    continue
                if part.type == ContentPartTypeEnum.image_url:
                    media, media_type = part.image_url, MMUrlType.IMAGE
                    kind = "image"
                elif part.type == ContentPartTypeEnum.video_url:
                    media, media_type = part.video_url, MMUrlType.VIDEO
                    kind = "video"
                else:
                    raise FtRuntimeException(
                        ExceptionType.INVALID_PARAMS,
                        f"Unsupported M3.1 VL content type: {part.type}",
                    )
                if media is None:
                    raise FtRuntimeException(
                        ExceptionType.INVALID_PARAMS, f"Missing {kind} URL"
                    )
                urls.append(media.url)
                types.append(media_type)
                config = (
                    part.preprocess_config.model_copy()
                    if part.preprocess_config is not None
                    else MMPreprocessConfigPart()
                )
                for field in ("max_long_side_pixel", "fps"):
                    value = getattr(media, field, None)
                    if value is not None:
                        setattr(config, field, value)
                    value = getattr(part, field, None)
                    if value is not None:
                        setattr(config, field, value)
                preprocess_configs.append(get_preprocess_config(config))
                content.append({"type": kind, kind: media.url})
            rendered_message["content"] = content

        return self._render_request(
            request, request_dict, urls, types, preprocess_configs
        )


register_renderer("minimax_m31", MiniMaxM31Renderer)
register_renderer("minimax_m31_vl", MiniMaxM31VLRenderer)
