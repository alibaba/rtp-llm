import copy
import json
import logging
from typing import List, Optional

from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ContentPartTypeEnum,
    MMPreprocessConfigPart,
)
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RenderedInputs,
    RendererParams,
)
from rtp_llm.utils.multimodal_util import MMPreprocessConfig, MMUrlType, MultimodalInput

class CustomModalProxyRenderer(CustomChatRenderer):
    """
    A proxy renderer that wraps an existing CustomChatRenderer to inject
    custom multimodal data processing capabilities without modifying the
    original renderer's logic.
    """
    def __init__(self, tokenizer: BaseTokenizer, renderer_params: RendererParams, wrapped_renderer: CustomChatRenderer):
        super().__init__(tokenizer, renderer_params)
        self.wrapped_renderer = wrapped_renderer
        logging.info(f"CustomModalProxyRenderer wrapping {type(wrapped_renderer).__name__}")

    def _create_mm_preprocess_config(self, part_config: Optional[MMPreprocessConfigPart]) -> MMPreprocessConfig:
        """Helper to convert MMPreprocessConfigPart to MMPreprocessConfig."""
        if not part_config:
            return MMPreprocessConfig()
        return MMPreprocessConfig(
            width=part_config.resized_width or -1,
            height=part_config.resized_height or -1,
            min_pixels=part_config.min_pixels or -1,
            max_pixels=part_config.max_pixels or -1,
            fps=part_config.fps or -1,
            min_frames=part_config.min_frames or -1,
            max_frames=part_config.max_frames or -1,
        )

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        modified_request = copy.deepcopy(request)
        
        # This list will hold the final ordered MultimodalInputs. 
        # We use None as a placeholder for inputs that the base renderer will handle (Images/Videos).
        final_multimodal_inputs: List[Optional[MultimodalInput]] = []
        
        for message in modified_request.messages:
            if isinstance(message.content, list):
                new_content_list = []
                for part in message.content:
                    if part.type == ContentPartTypeEnum.custom:
                        serialized_data = ""
                        mm_type = MMUrlType.DEFAULT
                        
                        try:
                            if isinstance(part.data, str):
                                # If it's a string, try to load it to verify if it's valid JSON
                                json.loads(part.data)
                                serialized_data = part.data
                            else:
                                # For dict, list, or primitives, try to dump it to JSON string
                                serialized_data = json.dumps(part.data)
                            mm_type = MMUrlType.CUSTOM_JSON
                        except (ValueError, TypeError):
                            # Fallback to Raw if not valid JSON or not serializable
                            serialized_data = str(part.data)
                            mm_type = MMUrlType.CUSTOM_RAW

                        mm_input = MultimodalInput(
                            url=serialized_data,
                            mm_type=mm_type,
                            config=self._create_mm_preprocess_config(part.preprocess_config)
                        )
                        final_multimodal_inputs.append(mm_input)
                    
                    elif part.type in [ContentPartTypeEnum.image_url, ContentPartTypeEnum.video_url, ContentPartTypeEnum.audio_url, ContentPartTypeEnum.igraph]:
                        # Mark a spot for base renderer's output
                        final_multimodal_inputs.append(None)
                        new_content_list.append(part)
                    else:
                        # Text or other types do not consume multimodal slots
                        new_content_list.append(part)
                message.content = new_content_list
        base_rendered_inputs = self.wrapped_renderer.render_chat(modified_request)
        base_inputs_iter = iter(base_rendered_inputs.multimodal_inputs)

        # Fill in the placeholders with actual inputs from base renderer
        # Note: We reconstruct the list to satisfy type checking (remove Optional)
        completed_inputs: List[MultimodalInput] = []
        for item in final_multimodal_inputs:
            if item is None:
                try:
                    completed_inputs.append(next(base_inputs_iter))
                except StopIteration:
                    logging.error("Base renderer produced fewer multimodal inputs than expected.")
            else:
                completed_inputs.append(item)
        
        # Append any remaining inputs from base renderer (just in case)
        for remaining in base_inputs_iter:
            logging.warning(f"Base renderer produced extra input: {remaining.url}")
            completed_inputs.append(remaining)

        # Return a new RenderedInputs object with the combined multimodal_inputs
        # RenderedInputs constructor expects separate lists for urls, types, and configs
        final_urls = [i.url for i in completed_inputs]
        final_types = [i.mm_type for i in completed_inputs]
        final_configs = [i.config for i in completed_inputs]

        return RenderedInputs(
            input_ids=base_rendered_inputs.input_ids,
            rendered_prompt=base_rendered_inputs.rendered_prompt,
            input_urls=final_urls,
            input_urls_type=final_types,
            preprocess_configs=final_configs,
        )