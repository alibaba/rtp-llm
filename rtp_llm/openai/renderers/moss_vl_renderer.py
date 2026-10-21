from rtp_llm.models.moss_vl import (
    MOSS_VL_IMAGE_PLACEHOLDER,
    MOSS_VL_VIDEO_PLACEHOLDER,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.qwen_vl_renderer import Qwen2VLRenderer

# The MOSS-VL chat template emits plain-text media placeholders that the HF
# processor (not the tokenizer) expands. In RTP the placeholder-to-token
# substitution happens here so the tokenizer emits real marker ids:
# - image -> ONE <|image_pad|> marker (no vision_start/end wrapper)
# - video  -> <|vision_start|><|video_pad|><|vision_end|> (single video
#   marker; the engine marker mode expands it to per-frame image markers
#   with timestamp text once the frame count is known).
_IMAGE_TOKEN = "<|image_pad|>"
_VIDEO_SPAN = "<|vision_start|><|video_pad|><|vision_end|>"


class MossVLRenderer(Qwen2VLRenderer):
    def _render_messages(self, request, add_vision_id):
        result = super()._render_messages(request, add_vision_id)
        prompt = result.prompt
        if MOSS_VL_IMAGE_PLACEHOLDER in prompt:
            prompt = prompt.replace(MOSS_VL_IMAGE_PLACEHOLDER, _IMAGE_TOKEN)
        if MOSS_VL_VIDEO_PLACEHOLDER in prompt:
            prompt = prompt.replace(MOSS_VL_VIDEO_PLACEHOLDER, _VIDEO_SPAN)
        result.prompt = prompt
        return result


register_renderer("moss_vl", MossVLRenderer)
