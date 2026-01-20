import copy
import os
from typing import List, Optional

from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase

from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.models.multimodal.multimodal_util import MMUrlType
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    ContentPartTypeEnum,
    RoleEnum,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import PromptWithMMInput
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RenderedInputs,
    RendererParams,
)
from rtp_llm.openai.renderers.llava_renderer import Conversation, SeparatorStyle

roles = {
    RoleEnum.user: "Human",
    RoleEnum.assistant: "Assistant",
    RoleEnum.system: "System",
}

conv_turing_v1 = Conversation(
    system_content="",
    roles={
        RoleEnum.user: "\n\nHuman",
        RoleEnum.assistant: "\n\nAssistant",
        RoleEnum.system: "\n\nSystem",
    },
    sep_style=SeparatorStyle.TWO,
    seps=["", "<|endoftext|>"],
    connector=[": ", ":"],
)

conv_chatml = Conversation(
    system_content="<|im_start|>system\nA conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.",
    roles={
        RoleEnum.user: "<|im_start|>user\n",
        RoleEnum.assistant: "<|im_start|>assistant\n",
        RoleEnum.system: "<|im_start|>system\n",
    },
    sep_style=SeparatorStyle.SINGLE,
    seps=["<|im_end|>\n"],
    connector=["", ""],
)

emb_chatml = Conversation(
    system_content="<|im_start|>system\nA conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers. ",
    roles={
        RoleEnum.user: "<|im_start|>user\n",
        RoleEnum.assistant: "<|im_start|>assistant\n",
        RoleEnum.system: "<|im_start|>system\n",
    },
    sep_style=SeparatorStyle.SINGLE,
    seps=["<|im_end|>\n"],
    connector=["", "<|im_end|>"],
    image_sep="\n<image>",
)


class FlotVLRenderer(CustomChatRenderer):
    def __init__(self, tokenizer: BaseTokenizer, renderer_params: RendererParams):
        super().__init__(tokenizer, renderer_params)

    def _render_messages(self, messages: List[ChatMessage]) -> PromptWithMMInput:
        return conv_turing_v1.render_messages(messages, self.tokenizer)

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        messages = copy.deepcopy(request.messages)
        prompt_and_mm_input = self._render_messages(messages)
        input_ids = self.tokenizer.encode(prompt_and_mm_input.prompt)
        return RenderedInputs(
            input_ids=input_ids,
            input_urls=prompt_and_mm_input.urls,
            rendered_prompt=prompt_and_mm_input.prompt,
        )


class TbstarsVL004VLRenderer(FlotVLRenderer):
    def _render_messages(self, messages: List[ChatMessage]) -> PromptWithMMInput:
        rendered_res = conv_chatml.render_messages(messages, self.tokenizer)
        prompt = ""
        image_tag_pair = "<image></image>"
        for splited_prompt in rendered_res.prompt.split("<image>"):
            prompt = prompt + splited_prompt + image_tag_pair
        rendered_res.prompt = prompt[: -len(image_tag_pair)]
        return rendered_res


class BiEncoderVLTbstarsRenderer(FlotVLRenderer):
    def _render_messages(self, messages: List[ChatMessage]) -> PromptWithMMInput:
        rendered_res = emb_chatml.render_messages(messages, self.tokenizer)
        prompt = ""
        image_tag_pair = "<image></image>"
        for splited_prompt in rendered_res.prompt.split("<image>"):
            prompt = prompt + splited_prompt + image_tag_pair
        rendered_res.prompt = prompt[: -len(image_tag_pair)]
        return rendered_res


class MixtbstarsRenderer(CustomChatRenderer):
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, renderer_params: RendererParams
    ):
        super().__init__(tokenizer, renderer_params)


class MixtbstarsMixedInputRenderer(CustomChatRenderer):
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, renderer_params: RendererParams
    ):
        super().__init__(tokenizer, renderer_params)
        self.search_dom = os.environ.get(
            "IGRAPH_SEARCH_DOM", "com.taobao.search.igraph.common"
        )

    def _render_messages(
        self, messages: List[ChatMessage], search_dom: Optional[str]
    ) -> PromptWithMMInput:
        urls = []
        types = []
        preprocess_configs = []
        final_messages = []
        for message in messages:
            if isinstance(message.content, str):
                final_messages.append(
                    {"role": message.role.value, "content": message.content}
                )
            elif isinstance(message.content, list):
                now_message = {"role": message.role.value}
                now_content = []
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        assert isinstance(content_part.text, str)
                        now_content.append({"type": "text", "text": content_part.text})
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert content_part.image_url != None
                        urls.append(content_part.image_url.url)
                        types.append(MMUrlType.IMAGE)
                        now_content.append(
                            {"type": "image", "image": content_part.image_url.url}
                        )
                    elif content_part.type == ContentPartTypeEnum.igraph:
                        assert content_part.igraph != None
                        assert content_part.igraph.table_name != None
                        assert content_part.igraph.item_id != None
                        url = content_part.igraph.item_id
                        urls.append(url)
                        types.append(MMUrlType.IGRAPH)
                        now_content.append({"type": "igraph", "url": url})
                now_message["content"] = now_content
                final_messages.append(now_message)

        prompt = self.tokenizer.apply_chat_template(
            final_messages,
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=True,
        )

        return PromptWithMMInput(
            prompt=prompt,
            urls=urls,
            mm_types=types,
            preprocess_configs=preprocess_configs,
        )

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        messages = copy.deepcopy(request.messages)
        if request.extend_fields:
            search_dom = request.extend_fields.get("igraph_searchdom", None)
        else:
            search_dom = None
        if not search_dom:
            search_dom = self.search_dom
        prompt_and_mm_input = self._render_messages(messages, search_dom)
        input_ids = self.tokenizer.encode(prompt_and_mm_input.prompt)
        return RenderedInputs(
            input_ids=input_ids,
            input_urls=prompt_and_mm_input.urls,
            rendered_prompt=prompt_and_mm_input.prompt,
            input_urls_type=prompt_and_mm_input.mm_types,
            preprocess_configs=prompt_and_mm_input.preprocess_configs,
        )


register_renderer("turing_005_vl", FlotVLRenderer)
register_renderer("tbstars_vl_002", FlotVLRenderer)
register_renderer("tbstars_vl_004", TbstarsVL004VLRenderer)
register_renderer("tbstars_vl_008o", TbstarsVL004VLRenderer)
register_renderer("biencoder_vl_tbstars", BiEncoderVLTbstarsRenderer)
register_renderer("mixtbstars", MixtbstarsMixedInputRenderer)
