"""MiniMax-M3.1 vision-language chat input renderer."""

import json

from rtp_llm.config.generate_config import ThinkingMode
from rtp_llm.openai.api_datatype import ContentPartTypeEnum, MMPreprocessConfigPart
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import BasicRenderer
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs
from rtp_llm.openai.renderers.llava_renderer import get_preprocess_config
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.minimax_m31_detector import (
    MiniMaxM31Detector,
)
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
        if generate_env_config.think_mode == "disabled":
            generate_env_config.think_mode = "adaptive"
        generate_env_config.think_start_tag = "<mm:think>"
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
        return request.resolve_thinking_mode(ThinkingMode.ADAPTIVE)

    def _create_detector(self, request):
        return MiniMaxM31Detector() if request.tools else None

    def _compile_jinja_template(self, chat_template):
        template = super()._compile_jinja_template(chat_template)

        def tojson(_eval_context, value, **kwargs):
            if isinstance(value, str):
                return value
            kwargs.setdefault("sort_keys", False)
            kwargs.setdefault("ensure_ascii", False)
            return json.dumps(value, **kwargs)

        template.environment.filters["tojson"] = tojson
        return template

    def _render_request(self, request, request_dict, urls=(), types=(), configs=()):
        for message in request_dict["messages"]:
            for call in message.get("tool_calls") or ():
                function = call.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    function["arguments"] = json.loads(arguments or "{}")
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
        if self.add_generation_prompt:
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
                    raise ValueError(f"Unsupported M3.1 VL content type: {part.type}")
                if media is None:
                    raise ValueError(f"Missing {kind} URL")
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
