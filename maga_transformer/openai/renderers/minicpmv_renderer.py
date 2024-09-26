import copy
from typing import List
from PIL import Image
from transformers import PreTrainedTokenizerBase, AutoProcessor

from maga_transformer.openai.api_datatype import (ChatMessage,
                                                  ChatCompletionRequest,
                                                  RoleEnum,
                                                  ContentPartTypeEnum)
from maga_transformer.openai.renderers.custom_renderer import (
    CustomChatRenderer, RendererParams, RenderedInputs)
from maga_transformer.openai.renderer_factory_register import register_renderer
from maga_transformer.utils.multimodal_util import (MMUrlType,
                                                    get_bytes_io_from_url)
from maga_transformer.models.minicpmv.minicpmv import encode_video


class MiniCPMVConversation():

    def render_messages(self, messages: List[ChatMessage]):
        copy_messages = copy.deepcopy(messages)
        urls: List[str] = []
        types: List[MMUrlType] = []
        msgs = []
        images = []
        if copy_messages[0].role == RoleEnum.system:
            assert isinstance(copy_messages[0].content, str)
            msgs = [{'role': 'system', 'content': copy_messages[0].content}]
            copy_messages = copy_messages[1:]

        for index, message in enumerate(copy_messages):
            assert message.role in [RoleEnum.user, RoleEnum.assistant]
            cur_msgs = []
            if index == 0:
                assert message.role == RoleEnum.user
            if isinstance(message.content, str):
                cur_msgs.append(message.content)
            elif isinstance(message.content, list):
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        assert (isinstance(content_part.text, str))
                        cur_msgs.append(content_part.text)
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert (content_part.image_url != None)
                        urls.append(content_part.image_url.url)
                        data = get_bytes_io_from_url(content_part.image_url.url)
                        data = Image.open(data).convert("RGB")
                        images.append(data)
                        cur_msgs.append("(<image>./</image>)")
                        types.append(MMUrlType.IMAGE)
                    elif content_part.type == ContentPartTypeEnum.video_url:
                        assert (content_part.video_url != None)
                        urls.append(content_part.video_url.url)
                        data = get_bytes_io_from_url(content_part.video_url.url)
                        data = encode_video(data)
                        images.extend(data)
                        cur_msgs.extend(["(<image>./</image>)" for _ in range(len(data))])
                        types.append(MMUrlType.VIDEO)
                msgs.append({
                    "role": message.role,
                    "content": "\n".join(cur_msgs)
                })
        return msgs, urls, types, images


class MiniCPMVRenderer(CustomChatRenderer):

    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 renderer_params: RendererParams):
        super().__init__(tokenizer, renderer_params)
        self.processor = AutoProcessor.from_pretrained(self.ckpt_path,
                                                       trust_remote_code=True)
        self.conv_template = MiniCPMVConversation()

    def _render_messages(self, messages: List[ChatMessage]) -> RenderedInputs:
        msgs, urls, types, images = self.conv_template.render_messages(
            messages)
        prompt = self.tokenizer.apply_chat_template(msgs,
                                                    tokenize=False,
                                                    add_generation_prompt=True)
        if not images:
            images = [[]]
        input_ids = self.processor(prompt,
                                   images,
                                   return_tensors="pt",
                                   max_length=8192)['input_ids']
        return RenderedInputs(input_ids=input_ids,
                              rendered_prompt=prompt,
                              input_urls=urls,
                              input_urls_type=types)

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        return self._render_messages(request.messages)


register_renderer('minicpmv', MiniCPMVRenderer)
