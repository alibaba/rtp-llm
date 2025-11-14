from typing import Any, Dict, List, Union

from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ContentPart,
    ContentPartTypeEnum,
    RoleEnum,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import BasicRenderer
from rtp_llm.openai.renderers.custom_renderer import CustomChatRenderer, RenderedInputs
from rtp_llm.utils.util import check_with_info


class QwenV2AudioRenderer(BasicRenderer):
    def __init__(self, *args: Any, **kwargs: Any):
        # BasicRenderer.__init__ will call CustomChatRenderer.__init__ with all args
        super().__init__(*args, **kwargs)
        self.chat_template = self._create_chat_template()

    def _create_chat_template(self):
        """
        This default vicuna template formats inputs in the form of a chat history. For each message in the chat history:
        * the template will output the role of the speaker followed by the content of the message.
        * content is a list of strings and audios.
        * If the content element is an audio, the template will output a sequence of <|AUDIO|> tokens

        Example:

        ```python
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
                {"type": "text", "text": "What's that sound?"},
            ]},
            {"role": "assistant", "content": "It is the sound of glass shattering."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"},
                {"type": "text", "text": "How about this one?"},
            ]},
        ]

        result = template.render(messages=messages, add_generation_prompt=True)
        ```
        """
        # fmt: off
        return (
            "{% set audio_count = namespace(value=0) %}"
            "{% for message in messages %}"
                "{% if loop.first and message['role'] != 'system' %}"
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "{% endif %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{% if message['content'] is string %}"
                    "{{ message['content'] }}<|im_end|>\n"
                "{% else %}"
                    "{% for content in message['content'] %}"
                        "{% if 'audio' in content or 'audio_url' in content %}"
                            "{% set audio_count.value = audio_count.value + 1 %}"
                            "Audio {{ audio_count.value }}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
                        "{% elif 'text' in content %}"
                            "{{ content['text'] }}"
                        "{% endif %}"
                    "{% endfor %}"
                    "<|im_end|>\n"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "<|im_start|>assistant\n"
            "{% endif %}"
        )
        # fmt: on

    def _render_content(self, message: Union[str, List[ContentPart]]):
        if isinstance(message, str):
            return message
        message_list: List[Dict[str, Any]] = []
        for content_part in message:
            if content_part.type not in [
                ContentPartTypeEnum.audio_url,
                ContentPartTypeEnum.text,
            ]:
                raise Exception(
                    f"unsupported content_part: {content_part.type} in message: {message}"
                )
            if content_part.type == ContentPartTypeEnum.text:
                message_list.append({"type": "text", "text": content_part.text})
            else:
                message_list.append(
                    {"type": "audio", "audio_url": content_part.audio_url.url}
                )
        return message_list

    def _extract_audio_urls(self, request: ChatCompletionRequest) -> List[str]:
        urls: List[str] = []
        for message in request.messages:
            if isinstance(message.content, list):
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.audio_url:
                        check_with_info(
                            content_part.audio_url is not None,
                            f"audio_url should not be none in {content_part}",
                        )
                        urls.append(content_part.audio_url.url)
        return urls

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        rendered_request = []
        for message in request.messages:
            if message.role not in [RoleEnum.system, RoleEnum.assistant, RoleEnum.user]:
                raise Exception(
                    f"unsupported role: {message.role} in QwenV2Audio request: {request}"
                )
            rendered_request.append(
                {
                    "role": message.role.value,
                    "content": self._render_content(message.content),
                }
            )
        encoded_text = self.tokenizer.apply_chat_template(
            rendered_request,
            add_generation_prompt=True,
            tokenize=False,
            chat_template=self.chat_template,
        )
        audio_urls = self._extract_audio_urls(request)
        input_ids = self.tokenizer.encode(encoded_text)
        return RenderedInputs(
            rendered_prompt=encoded_text, input_urls=audio_urls, input_ids=input_ids
        )


register_renderer("qwen_v2_audio", QwenV2AudioRenderer)
