import copy
from typing import List

from PIL import Image
from transformers import AutoProcessor

from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    ContentPartTypeEnum,
    RoleEnum,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RenderedInputs,
    RendererParams,
)
from rtp_llm.utils.multimodal_util import MMUrlType, get_bytes_io_from_url

try:
    from decord import VideoReader, cpu
except ModuleNotFoundError:
    VideoReader = None
    cpu = None


def encode_video(video_path, max_num_frames: int = 32):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")) for v in frames]
    return frames


class MiniCPMVConversation:

    def render_messages(self, messages: List[ChatMessage]):
        copy_messages = copy.deepcopy(messages)
        urls: List[str] = []
        types: List[MMUrlType] = []
        msgs = []
        images = []
        if copy_messages[0].role == RoleEnum.system:
            assert isinstance(copy_messages[0].content, str)
            msgs = [{"role": "system", "content": copy_messages[0].content}]
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
                        assert isinstance(content_part.text, str)
                        cur_msgs.append(content_part.text)
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert content_part.image_url != None
                        urls.append(content_part.image_url.url)
                        data = get_bytes_io_from_url(content_part.image_url.url, download_headers=self.download_headers)
                        data = Image.open(data).convert("RGB")
                        images.append(data)
                        cur_msgs.append("(<image>./</image>)")
                        types.append(MMUrlType.IMAGE)
                    elif content_part.type == ContentPartTypeEnum.video_url:
                        assert content_part.video_url != None
                        urls.append(content_part.video_url.url)
                        data = get_bytes_io_from_url(content_part.video_url.url, download_headers=self.download_headers)
                        data = encode_video(data)
                        images.extend(data)
                        cur_msgs.extend(
                            ["(<image>./</image>)" for _ in range(len(data))]
                        )
                        types.append(MMUrlType.VIDEO)
                msgs.append({"role": message.role, "content": "\n".join(cur_msgs)})
        return msgs, urls, types, images


class MiniCPMVRenderer(CustomChatRenderer):

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
        super().__init__(tokenizer, renderer_params, generate_env_config, render_config, ckpt_path, misc_config, vit_config)
        self.processor = AutoProcessor.from_pretrained(
            self.ckpt_path, trust_remote_code=True
        )
        # Get vit_config if available
        download_headers = vit_config.download_headers
        self.conv_template = MiniCPMVConversation(download_headers=download_headers)

    def _render_messages(self, messages: List[ChatMessage]) -> RenderedInputs:
        msgs, urls, types, images = self.conv_template.render_messages(messages)
        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        if not images:
            images = [[]]
        input_ids = self.processor(
            prompt, images, return_tensors="pt", max_length=8192
        )["input_ids"]
        return RenderedInputs(
            input_ids=input_ids,
            rendered_prompt=prompt,
            input_urls=urls,
            input_urls_type=types,
        )

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        return self._render_messages(request.messages)


register_renderer("minicpmv", MiniCPMVRenderer)
