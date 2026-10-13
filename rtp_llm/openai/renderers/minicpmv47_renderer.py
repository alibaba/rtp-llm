from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs
from rtp_llm.openai.renderers.qwen_vl_renderer import Qwen2VLRenderer

IMAGE_PAD_TOKEN = "<|image_pad|>"
# One <image>...</image> span per picture.  The HF processor emits
# "<image_id>N</image_id>\n<image>\n<pads+\n+slice markers>\n</image>\n"; the
# span interior is fully replaced by the image's concatenated crop features
# (thumbnail + slices) at embedding time, so a single pad suffices inside and
# only the image-id prefix (outside the span) survives as text tokens.


class MiniCPMV47Renderer(Qwen2VLRenderer):
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        prompt_and_mm_input = self._render_messages(
            request,
            request.extra_configs.add_vision_id if request.extra_configs else True,
        )
        prompt = prompt_and_mm_input.prompt
        num_pads = prompt.count(IMAGE_PAD_TOKEN)
        if num_pads != len(prompt_and_mm_input.urls):
            raise ValueError(
                f"MiniCPM-V-4.7 rendered prompt has {num_pads} image placeholders but "
                f"{len(prompt_and_mm_input.urls)} image urls"
            )
        parts = prompt.split(IMAGE_PAD_TOKEN)
        rendered_parts = [parts[0]]
        for image_index in range(num_pads):
            rendered_parts.append(
                f"<image_id>{image_index + 1}</image_id>\n<image>\n{IMAGE_PAD_TOKEN}\n</image>\n"
            )
            rendered_parts.append(parts[image_index + 1])
        prompt = "".join(rendered_parts)
        input_ids = self.tokenizer.encode(prompt)
        return RenderedInputs(
            input_ids=input_ids,
            input_urls=prompt_and_mm_input.urls,
            rendered_prompt=prompt,
            input_urls_type=prompt_and_mm_input.mm_types,
            preprocess_configs=prompt_and_mm_input.preprocess_configs,
        )


register_renderer("minicpmv4_7", MiniCPMV47Renderer)
